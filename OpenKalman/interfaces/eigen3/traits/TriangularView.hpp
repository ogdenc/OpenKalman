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
 * \brief Type traits as applied to native Eigen::TriangularView.
 */

#ifndef OPENKALMAN_EIGEN3_TRIANGULARVIEW_HPP
#define OPENKALMAN_EIGEN3_TRIANGULARVIEW_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace EGI = Eigen::internal;

  namespace interface
  {
    template<typename MatrixType, unsigned int Mode>
    struct IndexTraits<Eigen::TriangularView<MatrixType, Mode>>
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


    template<typename MatrixType, unsigned int Mode>
    struct Elements<Eigen::TriangularView<MatrixType, Mode>>
    {
      using scalar_type = scalar_type_of_t<MatrixType>;


      template<typename Arg>
      static scalar_type_of_t<Arg> get(const Arg& arg, Eigen::Index i, Eigen::Index j)
      {
        if ((i > j and (Mode & Eigen::Upper) != 0) or (i < j and (Mode & Eigen::Lower) != 0)) return 0;
        else return arg.coeff(i, j);
      }


  #ifdef __cpp_concepts
      template<typename Arg> requires ((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0)
  #else
      template<typename Arg, std::enable_if_t<((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0), int> = 0>
  #endif
      static Arg&& set(Arg&& arg, const scalar_type_of_t<Arg>& s, Eigen::Index i, Eigen::Index j)
      {
        arg.coeffRef(i, j) = s;
        return std::forward<Arg>(arg);
      }
    };


    template<typename M, unsigned int Mode>
    struct Dependencies<Eigen::TriangularView<M, Mode>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename EGI::traits<Eigen::TriangularView<M, Mode>>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return TriangularMatrix<equivalent_self_contained_t<M>, triangle_type_of_v<Arg>> {std::forward<Arg>(arg)};
      }
    };


    template<typename MatrixType, unsigned int Mode>
    struct SingleConstant<Eigen::TriangularView<MatrixType, Mode>>
    {
      const Eigen::TriangularView<MatrixType, Mode>& xpr;

      constexpr auto get_constant()
      {
        if constexpr (zero_matrix<MatrixType> or ((Mode & Eigen::ZeroDiag) != 0 and diagonal_matrix<MatrixType, Likelihood::maybe>))
          return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<MatrixType>, 0>{};
        else
          return std::monostate{};
      }

      constexpr auto get_constant_diagonal()
      {
        using Scalar = scalar_type_of_t<MatrixType>;
        constexpr auto b = has_dynamic_dimensions<MatrixType> ? Likelihood::maybe : Likelihood::definitely;

        if constexpr (not square_matrix<MatrixType, Likelihood::maybe>)
        {
          return std::monostate{};
        }
        else if constexpr ((Mode & Eigen::ZeroDiag) == 0 and Eigen3::eigen_Identity<MatrixType>)
        {
          return internal::ScalarConstant<b, Scalar, 1>{};
        }
        else if constexpr (((Mode & Eigen::UnitDiag) != 0 and
          (((Mode & Eigen::Upper) != 0 and triangular_matrix<MatrixType, TriangleType::lower, Likelihood::maybe>) or
            ((Mode & Eigen::Lower) != 0 and triangular_matrix<MatrixType, TriangleType::upper, Likelihood::maybe>))))
        {
          return internal::ScalarConstant<b, Scalar, 1>{};
        }
        else if constexpr ((Mode & Eigen::ZeroDiag) != 0 and
          (((Mode & Eigen::Upper) != 0 and triangular_matrix<MatrixType, TriangleType::lower, Likelihood::maybe>) or
            ((Mode & Eigen::Lower) != 0 and triangular_matrix<MatrixType, TriangleType::upper, Likelihood::maybe>)))
        {
          return internal::ScalarConstant<b, Scalar, 0>{};
        }
        else
        {
          return constant_diagonal_coefficient {xpr.nestedExpression()};
        }
      }
    };


    template<typename MatrixType, unsigned int Mode>
    struct TriangularTraits<Eigen::TriangularView<MatrixType, Mode>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular =
        (t == TriangleType::lower and ((Mode & Eigen::Lower) != 0 or triangular_matrix<MatrixType, TriangleType::lower, b>)) or
        (t == TriangleType::upper and ((Mode & Eigen::Upper) != 0 or triangular_matrix<MatrixType, TriangleType::upper, b>)) or
        (t == TriangleType::diagonal and triangular_matrix<MatrixType, (Mode & Eigen::Lower) ? TriangleType::upper : TriangleType::lower, b>) or
        (t == TriangleType::any and square_matrix<MatrixType, b>);

      static constexpr bool is_triangular_adapter = true;

      // make_triangular_matrix not included because TriangularView is already triangular if square.
    };


#ifdef __cpp_concepts
    template<typename MatrixType, unsigned int Mode> requires
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      real_axis_number<constant_coefficient<MatrixType>> or
      real_axis_number<constant_diagonal_coefficient<MatrixType>>
    struct HermitianTraits<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct HermitianTraits<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      real_axis_number<constant_coefficient<MatrixType>> or
      real_axis_number<constant_diagonal_coefficient<MatrixType>>>>
#endif
    {
      static constexpr bool is_hermitian = diagonal_matrix<MatrixType>;

      template<HermitianAdapterType t, typename Arg>
      static constexpr auto make_hermitian_adapter(Arg&& arg)
      {
        constexpr auto HMode = t == HermitianAdapterType::upper ? Eigen::Upper : Eigen::Lower;
        constexpr auto TriMode = (Mode & Eigen::Upper) != 0 ? Eigen::Upper : Eigen::Lower;
        if constexpr (HMode == TriMode)
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().template selfadjointView<HMode>());
        else
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().adjoint().template selfadjointView<HMode>());
      }
    };


    template<typename MatrixType, unsigned int Mode>
    struct Conversions<Eigen::TriangularView<MatrixType, Mode>>
    {
      template<typename Arg>
      static auto
      to_diagonal(Arg&& arg)
      {
        // If it is a column vector, the TriangularView wrapper doesn't matter, and otherwise, the following will thow an exception:
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
   * \internal
   * \brief Matrix traits for Eigen::TriangularView.
   */
  template<typename M, unsigned int Mode>
  struct MatrixTraits<Eigen::TriangularView<M, Mode>> : MatrixTraits<std::decay_t<M>>
  {
  private:

    using Scalar = typename EGI::traits<Eigen::TriangularView<M, Mode>>::Scalar;
    static constexpr auto rows = row_dimension_of_v<M>;
    static constexpr auto columns = column_dimension_of_v<M>;

    static constexpr TriangleType triangle_type = Mode & Eigen::Upper ? TriangleType::upper : TriangleType::lower;

  public:

    template<HermitianAdapterType storage_triangle =
      triangle_type == TriangleType::upper ? HermitianAdapterType ::upper : HermitianAdapterType ::lower, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template SelfAdjointMatrixFrom<storage_triangle, dim>;


    template<TriangleType triangle_type = triangle_type, std::size_t dim = rows>
    using TriangularMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template TriangularMatrixFrom<triangle_type, dim>;


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::native_eigen_matrix<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::TriangularView<std::remove_reference_t<Arg>, Mode>(arg);
    }

  };

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
