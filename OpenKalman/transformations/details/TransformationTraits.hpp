/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Defines traits relating to transformations.
 */

#ifndef OPENKALMAN_TRANSFORMATIONTRAITS_HPP
#define OPENKALMAN_TRANSFORMATIONTRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;


  namespace internal
  {
    /**
     * \internal
     * \brief Whether an object is a linearized function (with defined Jacobian and optionally Hessian functions).
     * \tparam T The function.
     * \tparam order The maximum order in which T's Taylor series is defined.
     */
#ifdef __cpp_concepts
    template<typename T, std::size_t order = 1>
#else
    template<typename T, std::size_t order = 1, typename = void>
#endif
    struct is_linearized_function : std::false_type {};
  }


  /**
   * \brief A linearized function (with defined Jacobian and optionally Hessian functions).
   * \details If order == 1, then the Jacobian is defined. If order == 2, then the Hessian is defined.
   * \tparam T The function.
   * \tparam order The maximum order in which T's Taylor series is defined.
   */
  template<typename T, std::size_t order = 1>
#ifdef __cpp_concepts
  concept linearized_function =
#else
  constexpr bool linearized_function =
#endif
    oin::is_linearized_function<std::decay_t<T>, order>::value;


  namespace internal
  {

#ifdef __cpp_concepts
    template<typename T> requires
      (std::is_member_function_pointer_v<decltype(&T::operator())> or std::is_function_v<T>)
    struct is_linearized_function<T, 0> : std::true_type {};
#else
    template<typename T>
    struct is_linearized_function<T, 0, std::enable_if_t<
      (std::is_member_function_pointer_v<decltype(&T::operator())> or std::is_function_v<T>)>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T> requires
      (std::is_member_function_pointer_v<decltype(&T::jacobian)> and linearized_function<T, 0>)
    struct is_linearized_function<T, 1> : std::true_type {};
#else
    template<typename T>
    struct is_linearized_function<T, 1, std::enable_if_t<
      (std::is_member_function_pointer_v<decltype(&T::jacobian)> and linearized_function<T, 0>)>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T> requires
      (std::is_member_function_pointer_v<decltype(&T::hessian)> and linearized_function<T, 1>)
    struct is_linearized_function<T, 2> : std::true_type {};
#else
    template<typename T>
    struct is_linearized_function<T, 2, std::enable_if_t<
      (std::is_member_function_pointer_v<decltype(&T::hessian)> and linearized_function<T, 1>)>> : std::true_type {};
#endif


    /**
     * \internal
     * \brief Get term <code>order</code> of the Taylor expansion of T.
     */
#ifdef __cpp_concepts
    template<std::size_t order = 1, typename T> requires (order > 0) and (order <= 2)
#else
    template<std::size_t order = 1, typename T, std::enable_if_t<(order > 0) and (order <= 2), int> = 0>
#endif
    static constexpr auto get_Taylor_term(const T& t)
    {
      if constexpr (order == 1)
      {
        return [&t](auto&& ...inputs) { return t.jacobian(std::forward<decltype(inputs)>(inputs)...); };
      }
      else if constexpr (order == 2)
      {
        return [&t](auto&& ...inputs) { return t.hessian(std::forward<decltype(inputs)>(inputs)...); };
      }
    }


  } // namespace internal


  namespace internal
  {
    /**
     * \internal
     * \brief The MatrixTraits of a noise perturbation.
     */
#ifdef __cpp_concepts
    template<typename T>
    struct PerturbationTraits;

    template<typename T> requires gaussian_distribution<T>
    struct PerturbationTraits<T> : MatrixTraits<typename DistributionTraits<T>::Mean> {};

    template<typed_matrix T>
    struct PerturbationTraits<T> : MatrixTraits<std::decay_t<T>> {};
#else
    template<typename T, typename = void>
    struct PerturbationTraits;

    template<typename T>
    struct PerturbationTraits<T, std::enable_if_t<gaussian_distribution<T>>>
      : MatrixTraits<typename DistributionTraits<T>::Mean> {};

    template<typename T>
    struct PerturbationTraits<T, std::enable_if_t<typed_matrix<T>>>
      : MatrixTraits<std::decay_t<T>> {};
#endif

  } // namespace internal


  /**
   * \brief T is an acceptable input to a tests.
   * \tparam Coeffs The expected coefficients of the tests input.
   */
  template<typename T, typename Coeffs = typename oin::PerturbationTraits<T>::RowCoefficients>
#ifdef __cpp_concepts
  concept transformation_input =
#else
  constexpr bool transformation_input =
#endif
    typed_matrix<T> and vector<T> and has_untyped_index<T, 1> and (not euclidean_transformed<T>) and
    equivalent_to<typename oin::PerturbationTraits<T>::RowCoefficients, Coeffs>;


  /**
   * \brief T is an acceptable noise perturbation input to a tests.
   * \tparam OutputCoefficients The expected coefficients of the tests output.
   */
#ifdef __cpp_concepts
  template<typename T, typename Coeffs = typename oin::PerturbationTraits<T>::RowCoefficients>
  concept perturbation = (gaussian_distribution<T> and
    equivalent_to<typename oin::PerturbationTraits<T>::RowCoefficients, Coeffs>) or transformation_input<T, Coeffs>;
#else
  namespace detail
  {
    template<typename T, typename Coeffs, typename = void>
    struct is_perturbation : std::false_type {};

    template<typename T, typename Coeffs>
    struct is_perturbation<T, Coeffs, std::enable_if_t<gaussian_distribution<T> and
      equivalent_to<typename oin::PerturbationTraits<T>::RowCoefficients, Coeffs>>> : std::true_type {};

    template<typename T, typename Coeffs>
    struct is_perturbation<T, Coeffs, std::enable_if_t<
      (not gaussian_distribution<T>) and transformation_input<T, Coeffs>>> : std::true_type {};
  }

  template<typename T, typename Coeffs = typename oin::PerturbationTraits<T>::RowCoefficients>
  constexpr bool perturbation = OpenKalman::detail::is_perturbation<T, Coeffs>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * \return If Arg is a distribution, return a random perturbation. Otherwise, return the fixed perturbation.
     */
#ifdef __cpp_concepts
    template<perturbation Arg>
#else
    template<typename Arg, std::enable_if_t<perturbation<Arg>, int> = 0>
#endif
    inline auto
    get_perturbation(Arg&& arg) noexcept
    {
      if constexpr(gaussian_distribution<Arg>)
        return std::forward<Arg>(arg)();
      else
        return std::forward<Arg>(arg);
    }
  }


  namespace detail
  {
    template<typename OutputCoefficients, typename In>
    inline auto zero_hessian_impl()
    {
      using InputCoefficients = vector_space_descriptor_of_t<In, 0>;
      constexpr std::size_t input_size = dimension_size_of_v<InputCoefficients>;
      constexpr std::size_t output_size = dimension_size_of_v<OutputCoefficients>;
      using HessianMatrixInBase = untyped_dense_writable_matrix_t<In, Layout::none, scalar_type_of_t<In>, input_size, input_size>;
      using HessianMatrixIn = Matrix<InputCoefficients, InputCoefficients, HessianMatrixInBase>;
      using HessianArrayIn = std::array<HessianMatrixIn, output_size>;

      HessianArrayIn a;
      a.fill(make_zero_matrix_like(a));
      return a;
    }
  }


  /// A tuple of zero-filled arrays of Hessian matrices, based on the input and each perturbation term.
  template<typename OutputCoefficients, typename In, typename ... Perturbations>
  inline auto zero_hessian()
  {
    static_assert(typed_matrix<In> and has_untyped_index<In, 1>);
    static_assert((perturbation<Perturbations> and ...));
    return std::tuple {OpenKalman::detail::zero_hessian_impl<OutputCoefficients, In>(),
      OpenKalman::detail::zero_hessian_impl<OutputCoefficients, Perturbations>()...};
  }


  /// A tuple of zero-filled arrays of Hessian matrices, based on the input and each perturbation term.
  template<typename OutputCoefficients, typename In, typename ... Perturbations>
  inline auto zero_hessian(In&&, Perturbations&&...)
  {
    return zero_hessian<OutputCoefficients, In, Perturbations...>();
  }

}


#endif //OPENKALMAN_TRANSFORMATIONTRAITS_HPP
