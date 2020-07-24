/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORMATION_H
#define OPENKALMAN_LINEARTRANSFORMATION_H


namespace OpenKalman
{

  /**
   * @brief A linear transformation from one single-column vector to another.
   *
   * @tparam InputCoefficients Coefficient types for the input.
   * @tparam OutputCoefficients Coefficient types for the output.
   * @tparam TransformationMatrix Transformation matrix. It is a native matrix type with rows corresponding to
   * OutputCoefficients and columns corresponding to InputCoefficients.
   * @tparam PerturbationTransformationMatrices Transformation matrices for each potential perturbation term.
   * if the parameter is not given, the transformation matrix is assumed to be identity (i.e., it is a translation).
   * It is a native matrix type with both rows and columns corresponding to OutputCoefficients.
   */
  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
  struct LinearTransformation;


  namespace internal
  {
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


  template<
    typename InputCoeffs,
    typename OutputCoeffs,
    typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
  struct LinearTransformation
  {
    using InputCoefficients = InputCoeffs;
    using OutputCoefficients = OutputCoeffs;
    static_assert(MatrixTraits<TransformationMatrix>::dimension == OutputCoefficients::size);
    static_assert(MatrixTraits<TransformationMatrix>::columns == InputCoefficients::size);
    static_assert(((MatrixTraits<PerturbationTransformationMatrices>::dimension == OutputCoefficients::size) and ...));
    static_assert(((MatrixTraits<PerturbationTransformationMatrices>::columns == OutputCoefficients::size) and ...));

    LinearTransformation(const TransformationMatrix& mat, const PerturbationTransformationMatrices& ... p_mats)
      : transformation_matrices(mat, p_mats...) {}

    template<typename T, typename ... Ps,
      std::enable_if_t<std::conjunction_v<is_typed_matrix<T>, is_typed_matrix<Ps>...>, int> = 0>
    LinearTransformation(const T& mat, const Ps& ... p_mats)
      : transformation_matrices(mat, p_mats...)
    {
      static_assert(is_equivalent_v<typename MatrixTraits<T>::RowCoefficients, OutputCoefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<T>::ColumnCoefficients, InputCoefficients>);
      static_assert(((is_equivalent_v<typename MatrixTraits<Ps>::RowCoefficients, OutputCoefficients>) and ...));
      static_assert(((is_equivalent_v<typename MatrixTraits<Ps>::ColumnCoefficients, OutputCoefficients>) and ...));
    }

    template<typename T, typename ... Ps,
      std::enable_if_t<std::conjunction_v<is_typed_matrix_base<T>, is_typed_matrix_base<Ps>...>, int> = 0>
    LinearTransformation(const T& mat, const Ps& ... p_mats)
      : transformation_matrices(make_Matrix<OutputCoefficients, InputCoefficients>(mat),
        make_Matrix<OutputCoefficients, InputCoefficients>(p_mats)...)
    {
      static_assert(MatrixTraits<T>::dimension == OutputCoefficients::size);
      static_assert(MatrixTraits<T>::columns == InputCoefficients::size);
      static_assert(((MatrixTraits<Ps>::dimension == OutputCoefficients::size) and ...));
      static_assert(((MatrixTraits<Ps>::columns == OutputCoefficients::size) and ...));
    }

  protected:
    const std::tuple<TypedMatrix<OutputCoefficients, InputCoefficients, TransformationMatrix>,
      TypedMatrix<OutputCoefficients, OutputCoefficients, PerturbationTransformationMatrices>...> transformation_matrices;

  private:
    template<std::size_t i, typename Input>
    constexpr decltype(auto) sumprod_term(Input&& input) const
    {
      constexpr auto mat_count = sizeof...(PerturbationTransformationMatrices) + 1;
      // If there is no corresponding transformation matrix, treat the transformation matrix as identity.
      if constexpr(i < mat_count)
        return std::get<i>(transformation_matrices) * std::forward<Input>(input);
      else
        return std::forward<Input>(input);
    }

    template<typename InputTuple, std::size_t...ints>
    constexpr auto sumprod(InputTuple&& inputs, std::index_sequence<ints...>) const
    {
      return strict((sumprod_term<ints>(std::get<ints>(std::forward<InputTuple>(inputs))) + ...));
    }

  public:
    template<
      typename In, typename ... Perturbations,
      std::enable_if_t<std::conjunction_v<is_typed_matrix<In>, is_perturbation<Perturbations>...>, int> = 0>
    auto operator()(In&& in, Perturbations&& ... perturbation) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert(is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, OutputCoefficients>...>);
      auto ret = sumprod(
        std::tuple {std::forward<In>(in), internal::get_perturbation(std::forward<Perturbations>(perturbation))...},
        std::make_index_sequence<sizeof...(Perturbations) + 1>{});
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(ret)>::RowCoefficients, OutputCoefficients>);
      return ret;
    }

    /// The Jacobians corresponding to the input and all perturbation matrices.
    /// Returns a tuple of the transformation matrices.
    template<typename In, typename ... Perturbations,
      std::enable_if_t<std::conjunction_v<is_typed_matrix<In>, is_perturbation<Perturbations>...>, int> = 0>
    auto jacobian(In&&, Perturbations&&...) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert(is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, OutputCoefficients>...>);

      constexpr auto mat_count = sizeof...(PerturbationTransformationMatrices);
      if constexpr(sizeof...(Perturbations) > mat_count)
      {
        constexpr auto size = sizeof...(Perturbations) - mat_count;
        constexpr auto dim = OutputCoefficients::dimension;
        using PStrict = typename MatrixTraits<TransformationMatrix>::template StrictMatrix<dim, dim>;
        auto id = MatrixTraits<PStrict>::identity();
        return std::tuple_cat(transformation_matrices, internal::tuple_replicate<size>(id));
      }
      else
      {
        return internal::tuple_slice<0, sizeof...(Perturbations) + 1>(transformation_matrices);
      }
    }

    /// The Hessian matrices corresponding to the input and all perturbation matrices.
    /// Returns a tuple of arrays of Hessian matrices. In this case, they are all zero.
    template<typename In, typename ... Perturbations,
      std::enable_if_t<std::conjunction_v<is_typed_matrix<In>, is_typed_matrix<Perturbations>...>, int> = 0>
    auto hessian(In&&, Perturbations&&...) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert(is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, OutputCoefficients>...>);
      constexpr std::size_t input_size = InputCoefficients::size;
      constexpr std::size_t output_size = OutputCoefficients::size;
      using HessianMatrixInBase = typename MatrixTraits<In>::template StrictMatrix<input_size, input_size>;
      using HessianMatrixIn = TypedMatrix<InputCoefficients, InputCoefficients, HessianMatrixInBase>;
      using HessianArrayIn = std::array<HessianMatrixIn, output_size>;
      HessianArrayIn a;
      a.fill(HessianMatrixIn::zero());
      if constexpr (sizeof...(Perturbations) >= 1)
      {
        using HessianMatrixNoiseBase = typename MatrixTraits<In>::template StrictMatrix<output_size, output_size>;
        using HessianMatrixNoise = TypedMatrix<OutputCoefficients, OutputCoefficients, HessianMatrixNoiseBase>;
        using HessianArrayNoise = std::array<HessianMatrixNoise, output_size>;
        HessianArrayNoise an;
        an.fill(HessianMatrixNoise::zero());
        return std::tuple_cat(std::tuple(std::move(a)), internal::tuple_replicate<sizeof...(Perturbations)>(std::move(an)));
      }
      else
      {
        return std::tuple(std::move(a));
      }
    }
  };


  /**
   * Deduction guides
   */

  template<typename T, typename ... Perturbations,
    std::enable_if_t<std::conjunction_v<is_typed_matrix<T>, is_perturbation<Perturbations>...>, int> = 0>
  LinearTransformation(T&&, Perturbations&& ...)
  -> LinearTransformation<
    typename MatrixTraits<T>::RowCoefficients,
    typename MatrixTraits<T>::ColumnCoefficients,
    typename MatrixTraits<T>::BaseMatrix,
    typename internal::PerturbationTraits<Perturbations>::BaseMatrix...>;

  template<typename T, typename ... Perturbations,
    std::enable_if_t<std::conjunction_v<is_typed_matrix_base<T>, is_typed_matrix_base<Perturbations>...>, int> = 0>
  LinearTransformation(T&&, Perturbations&& ...)
  -> LinearTransformation<
    Axes<MatrixTraits<T>::columns>,
    Axes<MatrixTraits<T>::dimension>,
    std::decay_t<T>,
    std::decay_t<Perturbations>...>;

}


#endif //OPENKALMAN_LINEARTRANSFORMATION_H
