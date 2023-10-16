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

      using type = std::tuple<typename Eigen::SelfAdjointView<MatrixType, UpLo>::MatrixTypeNested>;

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

      template<TriangleType t, typename Arg>
      static constexpr auto make_triangular_matrix(Arg&& arg)
      {
        constexpr auto TriMode = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        if constexpr (TriMode == UpLo)
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().template triangularView<TriMode>());
        else
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().adjoint().template triangularView<TriMode>());
      }

      static constexpr bool is_hermitian = (not complex_number<typename Eigen::internal::traits<MatrixType>::Scalar>) or
        real_axis_number<constant_coefficient<MatrixType>> or
        real_axis_number<constant_diagonal_coefficient<MatrixType>>;

      static constexpr HermitianAdapterType adapter_type =
        (UpLo & Eigen::Upper) != 0 ? HermitianAdapterType::upper : HermitianAdapterType::lower;

      // make_hermitian_adapter not included because SelfAdjointView is already hermitian if square.


#ifdef __cpp_concepts
      template<typename Arg> requires std::convertible_to<std::size_t, IndexType>
#else
      template<typename Arg, std::enable_if_t<indexible<Arg> and std::is_convertible_v<std::size_t, IndexType>, int> = 0>
#endif
      static constexpr decltype(auto) get(Arg&& arg, std::size_t i, std::size_t j)
      {
        using Scalar = scalar_type_of_t<MatrixType>;

        if constexpr (complex_number<Scalar>)
        {
          if ((i > j and (UpLo & Eigen::Upper) != 0) or (i < j and (UpLo & Eigen::Lower) != 0))
          {
            using std::conj;
            return Scalar {conj(std::as_const(arg).nestedExpression().coeff(static_cast<IndexType>(j), static_cast<IndexType>(i)))};
          }
          else return Scalar {std::as_const(arg).nestedExpression().coeff(static_cast<IndexType>(i), static_cast<IndexType>(j))};
        }
        else
        {
          if constexpr ((Eigen::internal::traits<Eigen::SelfAdjointView<MatrixType, UpLo>>::Flags & Eigen::LvalueBit) != 0)
          {
            if ((i > j and (UpLo & Eigen::Upper) != 0) or (i < j and (UpLo & Eigen::Lower) != 0))
              return std::forward<Arg>(arg).nestedExpression().coeffRef(static_cast<IndexType>(j), static_cast<IndexType>(i));
            else return std::forward<Arg>(arg).nestedExpression().coeffRef(static_cast<IndexType>(i), static_cast<IndexType>(j));
          }
          else
          {
            if ((i > j and (UpLo & Eigen::Upper) != 0) or (i < j and (UpLo & Eigen::Lower) != 0))
              return std::as_const(arg).nestedExpression().coeff(static_cast<IndexType>(j), static_cast<IndexType>(i));
            else return std::as_const(arg).nestedExpression().coeff(static_cast<IndexType>(i), static_cast<IndexType>(j));
          }
        }
      }


  #ifdef __cpp_concepts
      template<typename Arg> requires std::convertible_to<std::size_t, IndexType> and
        ((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0)
  #else
      template<typename Arg, std::enable_if_t<std::is_convertible_v<std::size_t, IndexType> and
        ((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0), int> = 0>
  #endif
      static void set(Arg& arg, const scalar_type_of_t<Arg>& s, std::size_t i, std::size_t j)
      {
        if ((i > j and (UpLo & Eigen::Upper) != 0) or (i < j and (UpLo & Eigen::Lower) != 0))
        {
          if constexpr (complex_number<scalar_type_of_t<MatrixType>>)
          {
            using std::conj;
            arg.coeffRef(static_cast<IndexType>(j), static_cast<IndexType>(i)) = conj(s);
          }
          else
            arg.coeffRef(static_cast<IndexType>(j), static_cast<IndexType>(i)) = s;
        }
        else arg.coeffRef(static_cast<IndexType>(i), static_cast<IndexType>(j)) = s;
      }
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
