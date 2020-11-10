/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TRANSFORMATIONTRAITS_HPP
#define OPENKALMAN_TRANSFORMATIONTRAITS_HPP

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


#ifdef __cpp_concepts
  /// T is a perturbation.
  template<typename T>
  concept perturbation = gaussian_distribution<T> or
    (typed_matrix<T> and column_vector<T> and not euclidean_transformed<T>);
#else
  template<typename T>
  struct is_perturbation : std::bool_constant<gaussian_distribution<T> or
    (typed_matrix<T> and column_vector<T> and not euclidean_transformed<T>)> {};

  /// Helper template for is_perturbation.
  template<typename T>
  inline constexpr bool is_perturbation_v = is_perturbation<T>::value;

  /// Helper template for is_perturbation.
  template<typename T>
  inline constexpr bool perturbation = is_perturbation<T>::value;
#endif


  namespace internal
  {
#ifdef __cpp_concepts
    template<typename Noise>
    struct PerturbationTraits;

    template<typename Noise> requires gaussian_distribution<Noise>
    struct PerturbationTraits<Noise> : MatrixTraits<typename DistributionTraits<Noise>::Mean> {};

    template<typed_matrix Noise>
    struct PerturbationTraits<Noise> : MatrixTraits<Noise> {};
#else
    template<typename Noise, typename Enable = void>
    struct PerturbationTraits;

    template<typename Noise>
    struct PerturbationTraits<Noise, std::enable_if_t<gaussian_distribution<Noise>>>
      : MatrixTraits<typename DistributionTraits<Noise>::Mean> {};

    template<typename Noise>
    struct PerturbationTraits<Noise, std::enable_if_t<typed_matrix<Noise>>>
      : MatrixTraits<Noise> {};
#endif


#ifdef __cpp_concepts
    template<perturbation Arg>
#else
    template<typename Arg, std::enable_if_t<is_perturbation_v<Arg>, int> = 0>
#endif
    inline auto
    get_perturbation(Arg&& arg) noexcept
    {
      if constexpr(gaussian_distribution<Arg>)
        return std::forward<Arg>(arg)();
      else
        return std::forward<Arg>(arg);
    }


    // In and Perburbations... are arguments to a transformation.
    template<typename In, typename ... Perturbations>
#ifdef __cpp_concepts
    concept transformation_args =
      ((MatrixTraits<In>::columns == 1) and ... and (internal::PerturbationTraits<Perturbations>::columns == 1)) and
      (equivalent_to<typename internal::PerturbationTraits<Perturbations>::RowCoefficients,
        typename MatrixTraits<In>::RowCoefficients> and ...);
#else
    inline constexpr bool transformation_args =
      ((MatrixTraits<In>::columns == 1) and ... and (internal::PerturbationTraits<Perturbations>::columns == 1)) and
      (equivalent_to<typename internal::PerturbationTraits<Perturbations>::RowCoefficients,
        typename MatrixTraits<In>::RowCoefficients> and ...);
#endif

  } // namespace internal


  namespace detail
  {
    template<typename OutputCoefficients, typename In>
    inline auto zero_hessian_impl()
    {
      using InputCoefficients = typename MatrixTraits<In>::RowCoefficients;
      constexpr std::size_t input_size = InputCoefficients::size;
      constexpr std::size_t output_size = OutputCoefficients::size;
      using HessianMatrixInBase = native_matrix_t<In, input_size, input_size>;
      using HessianMatrixIn = Matrix<InputCoefficients, InputCoefficients, HessianMatrixInBase>;
      using HessianArrayIn = std::array<HessianMatrixIn, output_size>;

      HessianArrayIn a;
      a.fill(HessianMatrixIn::zero());
      return a;
    }
  }


  /// A tuple of zero-filled arrays of Hessian matrices, based on the input and each perturbation term.
  template<typename OutputCoefficients, typename In, typename ... Perturbations>
  inline auto zero_hessian()
  {
    static_assert(column_vector<In>);
    static_assert((perturbation<Perturbations> and ...));
    return std::tuple {detail::zero_hessian_impl<OutputCoefficients, In>(),
      detail::zero_hessian_impl<OutputCoefficients, Perturbations>()...};
  }


  /// A tuple of zero-filled arrays of Hessian matrices, based on the input and each perturbation term.
  template<typename OutputCoefficients, typename In, typename ... Perturbations>
  inline auto zero_hessian(In&&, Perturbations&&...)
  {
    return zero_hessian<OutputCoefficients, In, Perturbations...>();
  }

}


#endif //OPENKALMAN_TRANSFORMATIONTRAITS_HPP
