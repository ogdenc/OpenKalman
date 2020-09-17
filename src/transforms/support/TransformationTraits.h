/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TRANSFORMATIONTRAITS_H
#define OPENKALMAN_TRANSFORMATIONTRAITS_H

#include <type_traits>

namespace OpenKalman
{
  /// Whether an object is a linearized function (with defined Jacobian and optionally Hessian functions).
  template<typename T, std::size_t order = 1, typename Enable = void>
  struct is_linearized_function : std::false_type {};

  template<typename T, std::size_t order>
  struct is_linearized_function<T&, order> : is_linearized_function<T, order> {};

  template<typename T, std::size_t order>
  struct is_linearized_function<T&&, order> : is_linearized_function<T, order> {};

  template<typename T, std::size_t order>
  struct is_linearized_function<const T, order> : is_linearized_function<T, order> {};

  /// Helper template for is_linearized_function.
  template<typename T, std::size_t order>
  inline constexpr bool is_linearized_function_v = is_linearized_function<T, order>::value;

  template<typename T>
  struct is_linearized_function<T, 0,
    std::enable_if_t<
      not std::is_reference_v<T> and not std::is_const_v<T> and
      (std::is_member_function_pointer_v<decltype(&std::decay_t<T>::operator())> or
      std::is_function_v<T>)>>
    : std::true_type {};

  template<typename T>
  struct is_linearized_function<T, 1,
    std::enable_if_t<
      not std::is_reference_v<T> and not std::is_const_v<T> and
      (std::is_member_function_pointer_v<decltype(&T::jacobian)> and
      is_linearized_function_v<T, 0>)>>
    : std::true_type
  {
    static constexpr auto get_lambda(const T& t)
    {
      return [&t] (auto&&...inputs) { return t.jacobian(std::forward<decltype(inputs)>(inputs)...); };
    }
  };

  template<typename T>
  struct is_linearized_function<T, 2,
    std::enable_if_t<
      not std::is_reference_v<T> and not std::is_const_v<T> and
      (std::is_member_function_pointer_v<decltype(&T::hessian)> and
      is_linearized_function_v<T, 1>)>>
    : std::true_type
  {
    static constexpr auto get_lambda(const T& t)
    {
      return [&t] (auto&&...inputs) { return t.hessian(std::forward<decltype(inputs)>(inputs)...); };
    }
  };


  template<typename T>
  struct is_perturbation : std::integral_constant<bool, is_Gaussian_distribution_v<T> or
    (is_typed_matrix_v<T> and is_column_vector_v<T> and not is_Euclidean_transformed_v<T>)> {};

  /// Helper template for is_perturbation.
  template<typename T>
  inline constexpr bool is_perturbation_v = is_perturbation<T>::value;


  namespace internal
  {
    template<typename Noise, typename T = void, typename Enable = void>
    struct PerturbationTraits;

    template<typename Noise, typename T>
    struct PerturbationTraits<Noise, T, std::enable_if_t<is_Gaussian_distribution_v<Noise>>>
      : MatrixTraits<typename DistributionTraits<Noise>::Mean> {};

    template<typename Noise, typename T>
    struct PerturbationTraits<Noise, T, std::enable_if_t<is_typed_matrix_v<Noise>>>
      : MatrixTraits<Noise> {};

    template<typename Arg, std::enable_if_t<is_perturbation_v<Arg>, int> = 0>
    inline auto
    get_perturbation(Arg&& arg) noexcept
    {
      if constexpr(is_Gaussian_distribution_v<Arg>)
        return std::forward<Arg>(arg)();
      else
        return std::forward<Arg>(arg);
    }


    namespace detail
    {
      template<std::size_t begin, typename T, std::size_t... I>
      constexpr auto tuple_slice_impl(T&& t, std::index_sequence<I...>)
      {
        return std::forward_as_tuple(std::get<begin + I>(std::forward<T>(t))...);
      }
    }

    /// Return a subset of a tuple, given an index range.
    template<std::size_t index1, std::size_t index2, typename T>
    constexpr auto tuple_slice(T&& t)
    {
      static_assert(index1 <= index2, "Index range is invalid");
      static_assert(index2 <= std::tuple_size_v<std::decay_t<T>>, "Index is out of bounds");
      return detail::tuple_slice_impl<index1>(std::forward<T>(t), std::make_index_sequence<index2 - index1>());
    }


    /// Create a tuple that replicates a value.
    template<std::size_t N, typename T>
    constexpr auto tuple_replicate(const T& t)
    {
      if constexpr(N < 1)
      {
        return std::tuple {};
      }
      else
      {
        return std::tuple_cat(std::make_tuple(t), tuple_replicate<N - 1>(t));
      }
    }

  }

}


#endif //OPENKALMAN_TRANSFORMATIONTRAITS_H
