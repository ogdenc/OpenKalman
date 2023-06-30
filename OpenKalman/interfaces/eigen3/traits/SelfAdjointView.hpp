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

#ifndef OPENKALMAN_EIGEN3_SELFADJOINTVIEW_HPP
#define OPENKALMAN_EIGEN3_SELFADJOINTVIEW_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace EGI = Eigen::internal;


  namespace interface
  {
    template<typename MatrixType, unsigned int UpLo>
    struct IndexTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      static constexpr std::size_t max_indices = max_indices_of_v<MatrixType>;

      template<std::size_t N, typename Arg>
      static constexpr auto get_index_descriptor(const Arg& arg)
      {
        return OpenKalman::get_index_descriptor<N>(arg.nestedExpression());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<MatrixType, b>;
    };


    template<typename MatrixType, unsigned int UpLo>
    struct Elements<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      using scalar_type = scalar_type_of_t<MatrixType>;


      template<typename Arg>
      static constexpr decltype(auto) get(Arg&& arg, Eigen::Index i, Eigen::Index j)
      {
        using Scalar = scalar_type_of_t<MatrixType>;

        if constexpr (complex_number<Scalar>)
        {
          if ((i > j and (UpLo & Eigen::Upper) != 0) or (i < j and (UpLo & Eigen::Lower) != 0))
          {
            using std::conj;
            return Scalar {conj(std::as_const(arg).nestedExpression().coeff(j, i))};
          }
          else return Scalar {std::as_const(arg).nestedExpression().coeff(i, j)};
        }
        else
        {
          if constexpr ((Eigen::internal::traits<Eigen::SelfAdjointView<MatrixType, UpLo>>::Flags & Eigen::LvalueBit) != 0)
          {
            if ((i > j and (UpLo & Eigen::Upper) != 0) or (i < j and (UpLo & Eigen::Lower) != 0))
              return std::forward<Arg>(arg).nestedExpression().coeffRef(j, i);
            else return std::forward<Arg>(arg).nestedExpression().coeffRef(i, j);
          }
          else
          {
            if ((i > j and (UpLo & Eigen::Upper) != 0) or (i < j and (UpLo & Eigen::Lower) != 0))
              return std::as_const(arg).nestedExpression().coeff(j, i);
            else return std::as_const(arg).nestedExpression().coeff(i, j);
          }
        }
      }


  #ifdef __cpp_concepts
      template<typename Arg> requires ((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0)
  #else
      template<typename Arg, std::enable_if_t<((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0), int> = 0>
  #endif
      static void set(Arg& arg, const scalar_type_of_t<Arg>& s, Eigen::Index i, Eigen::Index j)
      {
        if ((i > j and (UpLo & Eigen::Upper) != 0) or (i < j and (UpLo & Eigen::Lower) != 0))
        {
          if constexpr (complex_number<scalar_type_of_t<MatrixType>>)
          {
            using std::conj;
            arg.coeffRef(j, i) = conj(s);
          }
          else
            arg.coeffRef(j, i) = s;
        }
        else arg.coeffRef(i, j) = s;
      }
    };


    template<typename M, unsigned int UpLo>
    struct Dependencies<Eigen::SelfAdjointView<M, UpLo>>
    {
      static constexpr bool has_runtime_parameters = false;

      using type = std::tuple<typename Eigen::SelfAdjointView<M, UpLo>::MatrixTypeNested>;

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
        return SelfAdjointMatrix<equivalent_self_contained_t<M>, t> {std::forward<Arg>(arg)};
      }
    };


    template<typename MatrixType, unsigned int UpLo>
    struct SingleConstant<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      const Eigen::SelfAdjointView<MatrixType, UpLo>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (not complex_number<scalar_type_of_t<MatrixType>>)
          return constant_coefficient{xpr.nestedExpression()};
        else if constexpr (constant_matrix<MatrixType, CompileTimeStatus::known, Likelihood::maybe>)
        {
          if constexpr (real_axis_number<constant_coefficient<MatrixType>>)
            return constant_coefficient{xpr.nestedExpression()};
          else return std::monostate{};
        }
        else return std::monostate{};
      }

      constexpr auto get_constant_diagonal()
      {
        using Scalar = scalar_type_of_t<MatrixType>;
        if constexpr (Eigen3::eigen_Identity<MatrixType>) return internal::ScalarConstant<Likelihood::definitely, Scalar, 1>{};
        else return constant_diagonal_coefficient {xpr.nestedExpression()};
      }
    };


    template<typename MatrixType, unsigned int UpLo>
    struct TriangularTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
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
    };


#ifdef __cpp_concepts
    template<typename MatrixType, unsigned int UpLo> requires
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      real_axis_number<constant_coefficient<MatrixType>> or
      real_axis_number<constant_diagonal_coefficient<MatrixType>>
    struct HermitianTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
    template<typename MatrixType, unsigned int UpLo>
    struct HermitianTraits<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      real_axis_number<constant_coefficient<MatrixType>> or
      real_axis_number<constant_diagonal_coefficient<MatrixType>>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr HermitianAdapterType adapter_type =
        (UpLo & Eigen::Upper) != 0 ? HermitianAdapterType::upper : HermitianAdapterType::lower;

      // make_hermitian_adapter not included because SelfAdjointView is already hermitian if square.

    };


    template<typename MatrixType, unsigned int UpLo>
    struct Conversions<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      template<typename Arg>
      static auto
      to_diagonal(Arg&& arg)
      {
        // If it is a column vector, the SelfAdjointView wrapper doesn't matter, and otherwise, the following will throw an exception:
        return OpenKalman::to_diagonal(nested_matrix(std::forward<Arg>(arg)));
      }


      template<typename Arg>
      static constexpr decltype(auto)
      diagonal_of(Arg&& arg)
      {
        if constexpr (not square_matrix<Arg>) if (get_index_dimension_of<0>(arg) != get_index_dimension_of<1>(arg))
          throw std::logic_error {"Argument of diagonal_of must be a square matrix; instead it has " +
            std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
            std::to_string(get_index_dimension_of<1>(arg)) + " columns"};

        // Note: we assume that the nested matrix reference is not dangling.
        return OpenKalman::diagonal_of(std::forward<Arg>(arg).nestedExpression());
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


  /**
   * \internal
   * \brief Matrix traits for Eigen::SelfAdjointView.
   */
  template<typename M, unsigned int UpLo>
  struct MatrixTraits<Eigen::SelfAdjointView<M, UpLo>> : MatrixTraits<std::decay_t<M>>
  {
  private:

    using Scalar = typename EGI::traits<Eigen::SelfAdjointView<M, UpLo>>::Scalar;
    static constexpr auto rows = row_dimension_of_v<M>;
    static constexpr auto columns = column_dimension_of_v<M>;

    static constexpr HermitianAdapterType storage_triangle = UpLo & Eigen::Upper ? HermitianAdapterType::upper : HermitianAdapterType::lower;

  public:

    template<HermitianAdapterType storage_triangle = storage_triangle, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template SelfAdjointMatrixFrom<storage_triangle, dim>;


    template<TriangleType triangle_type = storage_triangle ==
      HermitianAdapterType::upper ? TriangleType::upper : TriangleType::lower, std::size_t dim = rows>
    using TriangularMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template TriangularMatrixFrom<triangle_type, dim>;


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::native_eigen_matrix<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::SelfAdjointView<std::remove_reference_t<Arg>, UpLo>(arg);
    }

  };

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_SELFADJOINTVIEW_HPP
