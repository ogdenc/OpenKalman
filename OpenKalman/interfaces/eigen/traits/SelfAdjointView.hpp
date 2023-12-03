/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to native Eigen::SelfAdjointView types.
 */

#ifndef OPENKALMAN_EIGEN_SELFADJOINTVIEW_HPP
#define OPENKALMAN_EIGEN_SELFADJOINTVIEW_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace interface
  {
    template<typename MatrixType, unsigned int UpLo>
    struct indexible_object_traits<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
    private:

      using Xpr = Eigen::SelfAdjointView<MatrixType, UpLo>;
      using IndexType = typename MatrixType::Index;

    public:

      using scalar_type = scalar_type_of_t<MatrixType>;


      template<typename Arg>
      static constexpr auto get_index_count(const Arg& arg) { return OpenKalman::get_index_count(nested_matrix(arg)); }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
      {
        return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
      }


      using dependents = std::tuple<typename Xpr::MatrixTypeNested>;


      static constexpr bool has_runtime_parameters = false;


      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }


      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        constexpr auto t = hermitian_adapter_type_of_v<Arg>;
        return SelfAdjointMatrix<equivalent_self_contained_t<MatrixType>, t> {std::forward<Arg>(arg)};
      }


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        if constexpr (not complex_number<scalar_type_of_t<MatrixType>>)
          return constant_coefficient{arg.nestedExpression()};
        else if constexpr (constant_matrix<MatrixType, CompileTimeStatus::known, Likelihood::maybe>)
        {
          if constexpr (real_axis_number<constant_coefficient<MatrixType>>)
            return constant_coefficient{arg.nestedExpression()};
          else return std::monostate{};
        }
        else return std::monostate{};
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        using Scalar = scalar_type_of_t<MatrixType>;
        if constexpr (Eigen3::eigen_Identity<MatrixType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 1>{};
        else return constant_diagonal_coefficient {arg.nestedExpression()};
      }


      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;


      template<Likelihood b>
      static constexpr bool is_square = square_matrix<MatrixType, b>;


      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = diagonal_matrix<MatrixType, b>;


      static constexpr bool is_triangular_adapter = false;


      static constexpr bool is_hermitian = (not complex_number<typename Eigen::internal::traits<MatrixType>::Scalar>) or
        real_axis_number<constant_coefficient<MatrixType>> or
        real_axis_number<constant_diagonal_coefficient<MatrixType>>;


      static constexpr HermitianAdapterType hermitian_adapter_type =
        (UpLo & Eigen::Upper) != 0 ? HermitianAdapterType::upper : HermitianAdapterType::lower;

    };

  } // namespace interface


  /**
   * \brief Deduction guide for converting Eigen::SelfAdjointView to SelfAdjointMatrix
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_SelfAdjointView M>
#else
  template<typename M, std::enable_if_t<Eigen3::eigen_SelfAdjointView<M>, int> = 0>
#endif
  SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<nested_matrix_of_t<M>, hermitian_adapter_type_of_v<M>>;

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN_SELFADJOINTVIEW_HPP
