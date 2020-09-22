/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COEFFICIENTS_H
#define OPENKALMAN_COEFFICIENTS_H

#include <array>
#include <functional>
#include <numeric>

namespace OpenKalman
{

  /**
   * @brief A set of coefficient types to be associated with a variable.
   * The types should be instances of is_coefficient.
   * @tparam Cs The coefficients (e.g., Axis, Angle, anything that as an instance of is_coefficient).
   */
  template<typename ... Cs>
  struct Coefficients;

  template<>
  struct Coefficients<>
  {
    static constexpr std::size_t size = 0; ///<Number of coefficients.
    static constexpr std::size_t dimension = 0; ///<Number of coefficients when converted to Euclidian.
    static constexpr bool axes_only = true; ///<All the coefficients are of type Axis.

    template<typename ... Cnew>
    using Prepend = Coefficients<Cnew...>;

    template<typename ... Cnew>
    using Append = Coefficients<Cnew...>;

    template<std::size_t i>
    using Coefficient = Coefficients;

    template<std::size_t count>
    using Take = Coefficients;

    template<std::size_t count>
    using Discard = Coefficients;

    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    template<typename Scalar>
    using SetCoeff = std::function<void(const Scalar, const std::size_t)>;

    template<typename Scalar>
    using GetCoeffFunction = Scalar(*)(const GetCoeff<Scalar>&);

    template<typename Scalar>
    using SetCoeffFunction = void(*)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&);

    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, dimension>
      to_Euclidean_array = {};

    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, size>
      from_Euclidean_array = {};

    /// Array of functions that retrieve a wrapped value without using expensive trigonometric functions.
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, size>
      wrap_array_get = {};

    /// Array of functions that set a wrapped value without using expensive trigonometric functions.
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const SetCoeffFunction<Scalar>, size>
      wrap_array_set = {};
  };


  template<typename C, typename ... Ctail>
  struct Coefficients<C, Ctail ...>
  {
    static_assert((is_coefficient_v<C> and ... and is_coefficient_v<Ctail>));
    static constexpr std::size_t size = C::size + Coefficients<Ctail...>::size; ///<Number of coefficients.

    /// Number of coefficients when converted to Euclidian.
    static constexpr std::size_t dimension = C::dimension + Coefficients<Ctail...>::dimension;

    /// All the coefficients are of type Axis.
    static constexpr bool axes_only = C::axes_only and Coefficients<Ctail...>::axes_only;

    template<typename ... Cnew>
    using Prepend = Coefficients<Cnew..., C, Ctail ...>;

    template<typename ... Cnew>
    using Append = Coefficients<C, Ctail ..., Cnew ...>;

    template<std::size_t i>
    using Coefficient = std::conditional_t<i == 0, C, typename Coefficients<Ctail...>::template Coefficient<i - 1>>;

    template<std::size_t count>
    using Take = std::conditional_t<count == 0,
      Coefficients<>,
      typename Coefficients<Ctail...>::template Take<count - 1>::template Prepend<C>>;

    template<std::size_t count>
    using Discard = std::conditional_t<count == 0,
      Coefficients,
      typename Coefficients<Ctail...>::template Discard<count - 1>>;

    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    template<typename Scalar>
    using SetCoeff = std::function<void(const Scalar, const std::size_t)>;

    template<typename Scalar>
    using GetCoeffFunction = Scalar(*)(const GetCoeff<Scalar>&);

    template<typename Scalar>
    using SetCoeffFunction = void(*)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&);

    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, dimension>
      to_Euclidean_array = internal::join(C::template to_Euclidean_array<Scalar, i>,
        Coefficients<Ctail...>::template to_Euclidean_array<Scalar, i + C::size>);

    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, size>
      from_Euclidean_array = internal::join(C::template from_Euclidean_array<Scalar, i>,
        Coefficients<Ctail...>::template from_Euclidean_array<Scalar, i + C::dimension>);

    /// Array of functions that retrieve a wrapped value without using expensive trigonometric functions.
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, size>
      wrap_array_get = internal::join(C::template wrap_array_get<Scalar, i>,
        Coefficients<Ctail...>::template wrap_array_get<Scalar, i + C::size>);

    /// Array of functions that set a wrapped value without using expensive trigonometric functions.
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const SetCoeffFunction<Scalar>, size>
      wrap_array_set = internal::join(C::template wrap_array_set<Scalar, i>,
        Coefficients<Ctail...>::template wrap_array_set<Scalar, i + C::size>);
  };

  /// Coefficients can, itself, be a coefficient. May be used to group coefficients together as a unit.
  template<typename...C>
  struct is_coefficient<Coefficients<C...>> : std::true_type {};

  template<typename...C>
  struct is_composite_coefficient<Coefficients<C...>> : std::true_type {};


  template<typename...C1, typename...C2>
  struct is_equivalent<Coefficients<C1...>, Coefficients<C2...>,
    std::enable_if_t<std::conjunction_v<is_equivalent<C1, C2>...> and (sizeof...(C1) != 1)>> : std::true_type {};

  template<typename T, typename U>
  struct is_equivalent<Coefficients<T>, Coefficients<U>> : is_equivalent<T, U> {};

  template<typename T, typename U>
  struct is_equivalent<T, Coefficients<U>> : is_equivalent<T, U> {};

  template<typename T, typename U>
  struct is_equivalent<Coefficients<T>, U> : is_equivalent<T, U> {};

  template<typename Ca, typename Cb, typename...C1, typename...C2>
  struct is_prefix<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>,
    std::enable_if_t<is_equivalent_v<Ca, Cb> and is_prefix_v<Coefficients<C1...>, Coefficients<C2...>> and
      not is_equivalent_v<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>>>> : std::true_type {};

  template<typename C>
  struct is_prefix<Coefficients<>, C,
    std::enable_if_t<is_coefficient_v<C> and not is_equivalent_v<Coefficients<>, C>>> : std::true_type {};

  template<typename C, typename...C1>
  struct is_prefix<C, Coefficients<C, C1...>,
    std::enable_if_t<is_coefficient_v<C>>> : std::true_type {};



  template<typename Coeffs, typename Scalar, std::enable_if_t<is_coefficient_v<Coeffs>, int> = 0>
  static Scalar to_Euclidean(const std::size_t row, const std::function<Scalar(const std::size_t)> get_coeff)
  {
    return Coeffs::template to_Euclidean_array<Scalar, 0>[row](get_coeff);
  }

  template<typename Coeffs, typename Scalar, std::enable_if_t<is_coefficient_v<Coeffs>, int> = 0>
  static Scalar from_Euclidean(const std::size_t row, const std::function<Scalar(const std::size_t)> get_coeff)
  {
    return Coeffs::template from_Euclidean_array<Scalar, 0>[row](get_coeff);
  }

  /// Wrap and return a single coefficient.
  template<typename Coeffs, typename F, std::enable_if_t<is_coefficient_v<Coeffs>, int> = 0>
  static auto wrap_get(const std::size_t row, const F& get_coeff)
  {
    static_assert(std::is_invocable_v<F, const std::size_t>);
    using Scalar = std::invoke_result_t<F, const std::size_t>;
    static_assert(std::is_arithmetic_v<Scalar>);
    return Coeffs::template wrap_array_get<Scalar, 0>[row](get_coeff);
  }

  /// Wrap and set a single coefficient.
  template<typename Coeffs, typename Scalar, typename FS, typename FG, std::enable_if_t<is_coefficient_v<Coeffs>, int> = 0>
  static void wrap_set(const Scalar s, const std::size_t row, const FS& set_coeff, const FG& get_coeff)
  {
    static_assert(std::is_invocable_v<FG, const std::size_t>);
    static_assert(std::is_invocable_v<FS, const std::size_t, const Scalar>);
    Coeffs::template wrap_array_set<Scalar, 0>[row](s, set_coeff, get_coeff);
  }


  namespace detail
  {
    template<typename C, std::size_t N, typename Enable = void>
    struct ReplicateImpl {};

    template<typename C>
    struct ReplicateImpl<C, 0, std::enable_if_t<is_coefficient_v<C>>>
    {
      using type = Coefficients<>;
    };

    template<typename C, std::size_t N>
    struct ReplicateImpl<C, N, std::enable_if_t<is_coefficient_v<C> and not is_composite_coefficient_v<C> and (N > 0)>>
    {
      using type = typename ReplicateImpl<C, N - 1>::type::template Append<C>;
    };

    template<typename...Cs, std::size_t N>
    struct ReplicateImpl<Coefficients<Cs...>, N, std::enable_if_t<(N > 0)>>
    {
      using type = typename ReplicateImpl<Coefficients<Cs...>, N - 1>::type::template Append<Cs...>;
    };
  }

  /**
   * @brief Create a Coefficients<...> alias in which the coefficients are C repeated N times.
   * @tparam C The coefficient to be repeated.
   * @tparam N The number of times to repeat coefficient C.
   */
  template<typename C, std::size_t N>
  using Replicate = typename detail::ReplicateImpl<C, N>::type;

  /**
   * Alias for a set of Axis coefficients of a given size.
   * @tparam size The number of Axes.
   */
  template<std::size_t size>
  using Axes = Replicate<Axis, size>;



  namespace internal
  {
    /**
     * @brief Concatenate any number of Coefficients<...> types.
     */
    template<typename ...>
    struct ConcatenateImpl;

    template<typename ... Cs1, typename ... Coeffs>
    struct ConcatenateImpl<Coefficients<Cs1...>, Coeffs...>
    {
      using type = typename ConcatenateImpl<Coeffs...>::type::template Prepend<Cs1...>;
    };

    template<typename Cs1, typename ... Coeffs>
    struct ConcatenateImpl<Cs1, Coeffs...>
    {
    using type = typename ConcatenateImpl<Coeffs...>::type::template Prepend<Cs1>;
    };

    template<>
    struct ConcatenateImpl<>
    {
      using type = Coefficients<>;
    };

  }

  /// Concatenate any number of Coefficients<...> types.
  template<typename ... Coeffs> using Concatenate = typename internal::ConcatenateImpl<Coeffs...>::type;


}// namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_H
